# NEMO SWG level-1 round 7 — thirty-day rung

Date: 2026-10-11. Base: `4ddd686dd`. Evidence directory:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/round7`.

## Preregistration

This section is frozen before the round-7 instrument is changed or any
legoESM value is compared with the admitted 30-day oracle record. The round-6
review corrections are closed at the base: its receipt citation gate passes
and its focused SWG battery is 194/194 green.

The admitted oracle is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/round6/oracle_swg_r6_30d`.
Its manifest stamps legoESM commit `011ba6c48`, and its admission states
`ADMITTED 30 STAGE-ENTRY RECORDS; PASSIVE OUTPUTS`. The compiled executed
branch is `SWG_OMIP_L1_P3D/BLD/ppsrc/nemo/`.

Frozen predictions and falsifiers:

1. **Independent month.** Starting only from the SWG card's shipped rest
   state and analytic shipped wind, normal production `step()` will execute
   720 one-hour steps on CPU in fp64. NEMO initializes U, V, T, S, and SSH to
   zero at compiled
   `SWG_OMIP_L1_P3D/BLD/ppsrc/nemo/usrdef_istate.f90:70-72,96`.
   Feeding any oracle state into legoESM, changing the deck, or evaluating a
   replay-only stage path **REFUTES** independence.
2. **Two admitted record kinds.** At the start of each shipped day, the
   instrument will reconstruct legoESM's complete 13-field SWE stage entry and
   score it against the admitted stage-1 record at kt
   1,25,...,697. The writer stores those exact 13 fields at compiled
   `SWG_OMIP_L1_P3D/BLD/ppsrc/nemo/stprk3.f90:146-161`. At the end of each day,
   it will score every dynamic/forcing array named by the existing restart
   reader against kt 24,48,...,720; NEMO writes the after state after the
   stage-3 `Naa -> Nbb` swap at compiled
   `SWG_OMIP_L1_P3D/BLD/ppsrc/nemo/stprk3.f90:324-340`, while the SWE forcing
   restart fields are written at compiled
   `SWG_OMIP_L1_P3D/BLD/ppsrc/nemo/sbcmod.f90:590-603`.
   Missing a record, field, day, or dtype **REFUTES** coverage.
3. **Prediction.** All 30 start-of-day stage boundaries and all 30 end-of-day
   restart boundaries are predicted bit-identical on every scored wet cell.
   Thus `first_day_over_bar` is predicted `None`, every exact field row is
   `AT-BAR`, and the worst maximum absolute difference is predicted zero.
   Any unequal scored bit **REFUTES** this prediction; it must be reported as
   `DEBT` even when its 1e-15 relative score would otherwise pass.
4. **Masks and precision.** T, U, V, and F fields will use the same card wet
   masks and shared staggered-face masks already certified by the ten-step
   ladder. Every oracle input and legoESM candidate must be float64, the JAX
   backend must be CPU, and the precision policy must be explicit fp64/libm.
   An all-ones mask, a T mask applied to a face field, or any float32 value
   **REFUTES** the measurement.
5. **Binding control.** After the clean first day, one admitted nonzero wet U
   input will be moved by exactly one representable fp64 ULP with `nextafter`.
   That branch will execute the normal production step, and at least one
   scored wet output must change and become `DEBT`. Perturbing a zero, changing
   an output after computation, or a plant that remains at the bar **REFUTES**
   the control.
6. **First non-bit statement.** If a daily row first leaves the bar, the
   receipt will name its day, record kind, field, first unequal cell, and
   maximum absolute difference. The daily acquisition can bracket a defect to
   the compiled three-stage program
   `SWG_OMIP_L1_P3D/BLD/ppsrc/nemo/stprk3.f90:141-340`; it cannot by itself
   assign a more precise source statement. A narrower attribution requires a
   preregistered per-step or per-term acquisition and remains UNATTRIBUTED
   until that exists. If every row is exact, there is no first non-bit
   statement.
7. **Rule 12.** The established ten-step rows may not move: given-entry
   338/338, chained 338/338, independent 50/50, and carry 29/29 remain exact.
   The month instrument extends the shared ladder only; it changes no model
   arithmetic, card, state, default, or configuration. Certified TSUNAMI,
   VORTEX, VORTEX_VEC, GYRE, LOCK_EXCHANGE, OVERFLOW, DINO, and ORCA2 cards
   therefore do not execute a changed statement. Any prior exact SWG row or
   certified-card hash that moves **REFUTES** landing.
8. **Gates.** The admitted-record provenance and schema must fail closed; the
   month arm and its plant must have direct tests; the receipt citation gate
   must map every P3D citation; and shifting a P3D citation by two lines must
   exit nonzero. The focused SWG battery runs on CPU in fp64. A green missing-
   record/schema plant, a nonbinding numerical plant, or a shifted citation
   that passes **REFUTES** the instrument.

No physical, configuration, threshold, cadence, or carried-state choice is
introduced. The user selected the shipped 720-step month and daily admitted
boundaries; the existing bit-for-bit bar, record fields, and masks are reused.

## Independent review

Independent review run by the operator.

## Measurement

PENDING — preregistration only.

## OPEN

1. Extend and calibrate the shared SWG ladder against both admitted daily
   record kinds, then run the clean and planted gates.
2. If the month is bit-identical, proceed next to the shipped 172800-step
   statistics; otherwise preregister the narrower acquisition required by the
   first daily boundary.
