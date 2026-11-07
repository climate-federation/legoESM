# Preregistration — round 183, land carried step-entry `rn2b`

Committed before any Round-183 baseline or candidate measurement.  Decision 61
authorizes the already measured Round-181 candidate to land with the two DINO
developed-state arms registered **UNMEASURED** because Round 182 proved that the
unchanged/control bridge destroys every prognostic family in its first step.
This round lands that candidate and does nothing else.  Repairing the DINO
developed-state bridge is explicitly deferred to Round 184.

## Compiled statement and one-variable candidate

The compiled GYRE stage computes `rab_b` and `rn2b` once from `Nbb`, copies
them to the current fields, and calls `zdf_phy(kstp,Nbb,Nbb,Nrhs)` at
`GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/stprk3.f90:159-168`.  It later
passes that same stored `rn2b` to `ldf_slp` at
`GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/stprk3.f90:173-178`.
The producer evaluates the active interior levels at
`GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/eosbn2.f90:1609-1619`.
The mixed-layer recurrence consumes `rn2b`, live `e3w_1d*(1+r3t)`, and the
strict below-threshold branch at
`GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/zdfmxl.f90:109-124`.

The one variable is the already existing GYRE card selector: change its
mixed-layer/slope N2 source from a second local recomputation to the existing
step-entry bundle.  The shared implementation accepts that bundle and skips
the dead recomputation.  No formula, threshold, timestep, restart schema,
stabilizer, or non-GYRE default changes.

## Frozen predictions and falsifiers

1. A same-tip baseline at `3c98979ae6` reproduces the immutable landed arm:
   first-over-bar kt=3; kt2 T/S/U/V remain at bar; day-30/day-240/day-360 T
   RMS equal `6.5725743747706026e-05`, `1.6448360701178680e-02`, and
   `1.1225660018551306e-02 K`.  A discrepancy is reconciled before direction
   is claimed.
2. Reapplying only the Round-181 implementation makes developed GYRE `nmln`
   and `hmlp` bit-exact under the complete production JIT step and eager
   execution.  Any unequal column refuses the landing.
3. The complete kt=1..10 ladder reproduces Round 181: kt=1 and kt=2 do not
   move, the first-over-bar remains kt=3, kt3 T/S are
   `4.940310525114455e-7 / 4.0085410546453204e-8`, and every moved row is in
   the registry.  An earlier first-over-bar or any kt=1 movement refuses it.
4. The month/year candidate reproduces, subject to same-tip measurement:
   day-30 `2.3276772050683987e-06 K`, day-240
   `6.5861718814795174e-05 K`, and day-360
   `2.6709923853294689e-03 K`.  Day 30, 240, and 360 must each improve.
5. The recipe-derived census finds exactly GYRE-zco plus both DINO Kamm cards
   selecting the carried route.  LOCK_EXCHANGE, OVERFLOW, ORCA2-zps, and the
   generic NEMO-GYRE recipe do not execute it.  Any additional executing card
   is measured before landing.
6. Both DINO from-rest card gates, both tanks, LOCK_EXCHANGE, OVERFLOW, and the
   generic card retain their certified behavior.  DINO developed-state is not
   inferred from these gates: it remains explicitly UNMEASURED under Decision
   61, with the unchanged-control Round-182 failure cited in the receipt.
7. The carried-route non-vacuity test must fail when the route is removed or a
   dead local N2 recomputation is restored.  The developed one-level plant
   must print `STATUS PLANT-FIRED` and exit nonzero.
8. The Decision-43/45 admission, citation gate, shifted-citation plant, and
   separate read-only Codex review all run at the final candidate commit.  A
   review verdict `DO NOT SHIP`, an unfired plant, a new test failure, or a
   dirty commit stamp refuses the landing.

No acquisition and no configuration decision are authorized.  The new
immutable before arm, if all gates pass, is the committed Round-183 candidate
artifact set.
