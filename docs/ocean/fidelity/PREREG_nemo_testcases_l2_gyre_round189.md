# Preregistration — round 189, inherited shortwave free-surface ratio

Committed before running any Round-189 substitution or producer score.
Round 188 proved every executed two-band shortwave statement BIT in isolated
eager and JIT execution on NEMO's operands, while the independent production
trace differs in 9,666 of 18,000 wet cells with maximum
`2.4678031493863273e-06 K`.  Its first non-bit input is the stage-3
`r3t(Kmm)` ratio: 600 of 600 wet columns, maximum
`1.6345230724468252e-09`.  Evidence for this round lives under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round189/`.

No model physics, public configuration, carried-state schema, default,
stabilizer, or NEMO source is authorized to change.  The round extends the
existing Round-186/188 shortwave gate and uses the same production
`LatLonCGridOceanModel.step -> self._step_jitted` path.  Isolated closure rows
remain labelled isolated and cannot establish production ownership.

## Compiled order

On the compiled HYB branch, stage 1 computes the after-level ratios from the
external-mode `ssha` by `CALL dom_qco_r3c_RK3` at
`GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/stprk3_stg.f90:172-192`.
Stage 2 forms the half-step `ssh(Kaa)` and `r3t(Kaa)` from the step-entry and
after levels at
`GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/stprk3_stg.f90:227-237`;
after the pointer rotation that value is stage 3's `Kmm`.  The T-point ratio
statement itself is `pr3t = pssh * r1_ht_0` at
`GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/domqco.f90:237-258`.
The shortwave call consumes `r3t(Kmm)` and associates its increment into
`Krhs` at
`GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/traqsr.f90:616-645`.

## Frozen experiment and predictions

1. Reproduce the Round-188 own-chain production-JIT shortwave row exactly:
   9,666/18,000 unequal cells and maximum
   `2.4678031493863273e-06 K`.  A mismatch means the one-variable experiment
   is not on the certified trajectory and stops the round.
2. Drive seed zero from rest through step 1079 with the ordinary production
   model.  At step 1080 branch once: the control retains the model's live
   stage-3 ratio; the candidate replaces only the ratio handed to the
   stage-3 shortwave evaluation with NEMO's recorded `r3t(Kmm)`.  Surface
   flux, reference geometry, masks, preceding accumulator, forcing, and every
   carried state leaf remain identical.  Both branches run through
   `self._step_jitted`.
3. Frozen attribution prediction: the NEMO-ratio arm removes at least 90% of
   the baseline maximum shortwave-row error.  This is **REFUTED** if the
   removed fraction is below 0.90 or if any registered pre-shortwave boundary
   moves.  Exact closure is predicted only conditionally, not assumed: a
   non-bit remainder may be the final `(Krhs + rate) - Krhs` association on
   the model's different preceding accumulator.
4. A production-path plant changes one active value of the injected NEMO
   ratio by one representable step.  It must move the stage-3 shortwave
   process boundary, print `STATUS PLANT-FIRED`, and exit nonzero.  A plant
   that fires only in an isolated copy is invalid.
5. If the NEMO-ratio arm makes the production-JIT shortwave row BIT, walk its
   producer in compiled order: step-entry ratio, after-level sea surface,
   `pr3t = pssh*r1_ht_0`, and the stage-2 half-step blend.  The first non-bit
   operand or statement owns the walk.  The model's producer must first
   reproduce its recorded `q_Kmm` bit-for-bit.
6. If the ratio arm does not close, do not walk upstream.  Rank the remaining
   shortwave inputs and the associated update one at a time, with the
   preceding accumulator reported explicitly; the first surviving boundary
   becomes the OPEN item.  No local shortwave statement may be blamed because
   Round 188 already proved the operator BIT on NEMO inputs.
7. No landing occurs without a one-variable NEMO-source-exact statement and
   the complete Decision-43/45/55/59 ladder, day-30, day-240, day-360,
   executing-card census, DINO, generic-card, tank, ORCA2-with-spec, plant,
   citation, and review gates.  A diagnostic-only result is HELD.

The final diff receives a separate read-only Codex review.  A `DO NOT SHIP`
verdict blocks any landing.  No configuration decision or acquisition is
expected.
