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

## Control erratum preregistration

This section was committed after the clean month measurement and the first
control attempt, but before changing or running the replacement control. The
original preregistration above remains frozen.

The first control selected the smallest-magnitude nonzero wet U value at the
exact admitted day-one boundary, moved it by one fp64 ULP, and ran one normal
production step. **REFUTED:** all scored SSH/U/V output bits were unchanged, so
the CLI correctly printed `REFUSE` and exited 2; that attempt is not a control.

The replacement keeps the preregistered field, boundary, one-ULP operation,
production path, masks, and scored outputs fixed. It selects the admitted wet
U value with the largest absolute magnitude, whose one-ULP displacement is the
largest absolute one-ULP U input available without changing the stipulated
perturbation. Prediction: at least one next-step wet SSH/U/V output changes and
the control exits 1 as `DEBT`. Unchanged output **REFUTES** the replacement too.

## Independent review

Independent review run by the operator.

## Measurement

Preregistration commit: `7a982cfbcf58190763ba0a0f6d76d985aad782ee`.
Committed month instrument: `ef3945bc687cd91b2d46adb93e09abc8dceae67e`.
Control erratum preregistration:
`2c1129ac605b8ac2938243207242427d858bf75a`.
Final measured control implementation:
`fbafd24d87175f5e66eff4e5daabc4a500a5f2c8`.

1. **CONFIRMED: the independent 30-day rung is bit-identical.** The card's
   shipped rest state and analytic wind ran 720 ordinary one-hour production
   steps. All 690 daily rows are exact: 390 rows from 30 complete 13-field
   start-of-day stage entries and 300 rows from 30 ten-field end-of-day
   restarts. `first_day_over_bar` and `first_kt_over_bar` are both `None`; the
   worst maximum absolute difference is zero. Artifact `swg_month.json`,
   SHA-256
   `8e23f7d21e5bf466ebc01b2a9a0175032eb1207f5372b83540ef4ae338b09a5a`.
2. **CONFIRMED: both record kinds are admitted and complete.** Before scoring,
   the gate reruns the committed checker over the instrumented and independent
   reference runs and requires exact equality with the saved admission. It
   therefore rechecks all 30 record hashes, all 30 passive restart pairs, and
   every ordinary T/U/V/W output variable. Saved admission SHA-256:
   `94db6dfea297fe828c55dc6c26fa78702cbbf9df399aa8f6793d252f33efae8e`.
   Missing files, unexpected files, wrong dtypes, nonfinite stored halos, or a
   changed acquisition commit refuse the month before any fidelity score.
3. **CONFIRMED: all native staggers and ratios remain exact.** Every T, U, V,
   and F field is scored with the card's certified native mask. This includes
   each daily `r3t/r3u/r3v/r3f`, current/before velocity, and current/before
   stress. Every candidate and oracle array is float64 under the explicit
   fp64/libm policy on CPU.
4. **REFUTED: the smallest-magnitude one-ULP U control binds.** The first
   committed control attempt changed one exact admitted nonzero wet day-one U
   input by one ULP, but the next production step returned identical scored
   SSH/U/V bits. It correctly printed `REFUSE` and exited 2. It is retained in
   the control erratum above and makes no fidelity claim.
5. **CONFIRMED: the corrected one-ULP control binds.** At admitted wet U cell
   `[1,74]`, the largest-magnitude day-one input moves by exactly one fp64 ULP,
   from `-0.027032120471695776` to `-0.027032120471695772` m/s. One normal
   production step changes one wet SSH cell, with maximum absolute difference
   `4.440892098500626e-16` m. The numerical relative score is below 1e-15, but
   the bit-exact gate correctly classifies the non-identical row `DEBT` and
   exits 1. Artifact `swg_month_ulp_plant.json`, SHA-256
   `b71b9f9cdaf8afac1bdb321b8935673ebceb79544a7da27bfd10dc5a904c600b`.

## First non-bit statement

There is none through 30 days. No daily stage-entry or restart field differs,
so the complete compiled three-stage program at
`SWG_OMIP_L1_P3D/BLD/ppsrc/nemo/stprk3.f90:141-340` remains bit-identical at
every admitted daily boundary. The planted row is a control, not a model-oracle
fidelity difference.

## Rule 12 disposition

No established exact row moved: given-entry is 338/338, chained is 338/338,
independent ten-step is 50/50, and carried-state is 29/29 bit-identical. Their
final artifacts have SHA-256 values
`1a87e47c468ba34bdad01658760ef3d451717af80e82e8c0e5ad483acc9393b9`,
`c458efee25f33f0dd4451a9708e723a99a25233b5d08200108c64327fda30a28`,
`fef9815261d71ca08e97e4376ab7302fa221d37cbc692a1625ce4fb0a19dd82c`,
and `73dff4e7d10455885bb5743079343512195a0f5e3eba127205653a277a8b1736`.
The established stage one-ULP plant still exits 1.

The base-to-tip production diff under `packages/` and `src/` is empty. Only
the shared fidelity ladder, its direct test, the citation map, and this receipt
change, so TSUNAMI, VORTEX, VORTEX_VEC, GYRE, LOCK_EXCHANGE, OVERFLOW, DINO,
and ORCA2 execute no changed model or card statement.

## Gates

The focused card, stage, differentiability, record, geometry, ladder, month,
and control battery passes 197/197; log SHA-256
`dc866d813a8f4361bdba76d3fcfd9455d9f163910fdc09e7cf2af358fb80a67c`.
Geometry remains 34/34 exact, and its signed-y Coriolis, wet-cell, and layer-
thickness plants each exit 1. The month control and established stage control
each exit 1. The receipt citation gate and shifted P3D citation plant are run
after this receipt is complete.

## Choices this round

No unasked physical, configuration, threshold, cadence, or carried-state
choice was made. Selecting the largest-magnitude nonzero wet U value changes
only the test control's ability to bind; the user required a one-ULP move of a
recorded nonzero consumed field. The original nonbinding choice remains
explicitly retracted.

## OPEN

1. The operator runs the independent review of round 7.
2. Preregister and score the shipped 172800-step statistics from the existing
   `round0_probe/year10` oracle record. No statistics claim is made here.
