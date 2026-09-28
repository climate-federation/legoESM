# Preregistration: NEMO-testcases L2 GYRE round 74 acquisition repair

Date: 2026-09-12. Frozen after the operator-reported round-73 acquisition
refusal and before editing or exercising its replacement acquisition card.

## Boundary and failed acquisition

Round 73 stopped at the first non-bit stage-1 transport input, `un_adv`:
580 of 580 wet U cells differ, with maximum absolute difference
`0.00012029895814569258`. The required kt=2 producer record is still absent.
The operator's attempt exited 65 before `makenemo` because the source-run
preflight searched for a resolved `nn_baro = 50` row that NEMO does not print.
This is an acquisition-instrument defect, not a scientific falsification and
not authority to edit production physics.

The R72 compiled program computes the live automatic external-step count into
`nn_e` and obtains `rDt_e = rn_Dt / REAL(nn_e,wp)` at
`GYRE_OMIP_L2_P3_SM_R72ZFOP/BLD/ppsrc/nemo/dynspg_ts.f90:1027-1045`. It prints
the resolved seconds and `nn_e` count at the same compiled source's
`:1062-1067`. The immutable R72 `ocean.output` contains `rDt_e = 288 s`,
`nn_e = 50`, and `ln_bt_auto=T`; it contains no resolved `nn_baro` row.

## Frozen repair and controls

Prepare a new round-74 acquisition card from the unchanged R72 source card,
with target configuration `GYRE_OMIP_L2_P3_SM_R74ADV2` and target evidence
directory `round74/oracle_advmean_kt2`. Reuse the already reviewed additive
round-73 `dynspg_ts.F90` patch, record gate, and admission gate. Do not change
the record layout, writer predicates, source configuration, namelist, or model
statement.

Replace only the nonexistent resolved-row assertion with checks for the exact
rows NEMO actually prints: the 14,400 s ocean step, 288 s external step,
50 `nn_e` iterations, and automatic-step selection. Retain the existing checks
for the ten-step horizon, assimilation cycle, tiling, and vector momentum
advection. The acquisition must continue to compare the copied target
`namelist_cfg` byte-for-byte with the source.

Prediction: the corrected preflight accepts the immutable R72 output and exits
zero without invoking `makenemo` or `mpirun`. A plant changing the logged
resolved `nn_e` count from 50 to 49 must exit nonzero and name the missing
`nn_e = 50` row. Shell parsing, additive patch dry-application, and the exact
preprocessed Fortran syntax proof must pass. Any failure records the repair as
REFUTED and no acquisition card is handed off.

If the operator later runs the full card, it must create a NEW target, emit the
exact 3,789,976-byte record, replay all 100 U/V entry-plus-increment rows and
both normalization rows bit-for-bit, and make stamp, header, truncation,
replay-ULP, and consumed-field plants exit nonzero. The commit stamp must equal
the clean committed checkout running the acquisition. No record exists in this
round, so the accumulator source walk remains UNREACHED.

## Source boundary and Rule 12

At each live external substep, compiled GYRE assigns `za2=wgtbtp2(jn)` and
left-associates the U/V accumulator statements at
`GYRE_OMIP_L2_P3_SM_R72ZFOP/BLD/ppsrc/nemo/dynspg_ts.f90:559-568`. It divides
the completed sums at `:797-801`, applies the active boundary exchange at
`:806-812`, and hands off the post-boundary values at `:813-817`. Those are the
next statements to walk after acquisition.

No production statement is eligible in this round. GYRE kt1--10 and days
1--30, LOCK_EXCHANGE, OVERFLOW, and DINO are UNREACHED because production is
unchanged. DINO retains explicit shared-accumulator and cancellation risk.
ORCA2 remains UNMEASURED WITH SPEC: resolve its compiled external-mode card;
record every substep entry, weight, transport, metric reciprocal, exit,
normalization, and boundary handoff for kt1--10; replay in compiled order;
register every moved row; preserve every AT-BAR row; and forbid an earlier
first-over-bar boundary.

No configuration/default, carried state, stabilizer, NEMO source/build/run,
year harness, reconciliation gate, freshwater pair, #1484 guard, or held
manifest may change. The round-70 patch remains held.
