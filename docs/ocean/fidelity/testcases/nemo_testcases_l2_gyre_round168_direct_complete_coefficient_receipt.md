# Round 168 receipt — direct complete-coefficient sensitivity

**Status: HELD.**  No production physics, configuration, carried state,
restart schema, or immutable before arm changed.  A private direct post-add
intervention closes Round 167's missing comparison: installing NEMO's recorded
complete tracer coefficient reduces the developed day-240 T3D RMS from
`1.584259320940647e-02` to `1.241263037891706e-03` K, removing
`1.460133017151476e-02` K (`92.1650261324976%`).  This is larger than the
heat-only arm's `1.315561886821479e-02` K (`83.03955478957635%`), so the
heat/TKE family is not the largest boundary.  The first remaining non-bit
operand read by the compiled matrix statement is its live `e3w(Kmm)` divisor.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round168/`.  The frozen
preregistration is commit `d2e022ccd`; the instrument and scientific run are
commit `40682ab53`.  The immutable production headline remains Round 163:
kt2 T/S/U/V AT-BAR, first-over-bar kt3, day-30 T RMS
`6.572574374770603e-05` K, day-240 `1.644836070117868e-02` K, and day-360
`1.122566001855131e-02` K.

## Compiled program and exact intervention

The compiled temperature branch forms `zwt = avt + ah_wslp2` at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:418-480`.
That same range proves the source order after the sum: the lower/upper
coefficients first divide `zwt` by live `e3w(Kmm)` at `:468-469`, and only
then does the diagonal read `e3t(Kaa)` at `:470`.  The LU, content, forward,
and backward recurrences are
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:527-582`.

The closure-to-heat path remains the selected TKE call, closure copy, and EVD
replacement in
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdfphy.f90:334-359`; the TKE
conversion to `avt_k` is
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdftke.f90:681-692`.
Round 168 does not change any of those statements.  It extends the existing
private two-array heat/viscosity seam with an optional third array applied
after the model's isoneutral addition and immediately before the production
matrix call.  Ordinary callers still pass no seam.  The two-array Round-167
form is unchanged.

The preregistration called this a temperature coefficient.  The exact runtime
scope is the shared tracer coefficient: this GYRE card has no distinct
double-diffusion salinity matrix, so production uses one coefficient for the
paired T/S solve.  The intervention is still one array at one compiled
boundary; it does not change viscosity, closure state, isoneutral computation,
geometry, content, forcing, or carried state.

## Calibration and controls

The Round-132 record audit passes all 360 daily boundaries and all 12 monthly
overlaps at the current commit.  Every Round-125 matrix field rebuild is BIT.
The free arm reproduces Round 167's step-1081 and day-240 values exactly.  The
direct identity arm supplies the free arm's own formed coefficient and differs
from free by zero state bytes through all 360 steps.

| step-1081 row | unequal / scored | maximum absolute | RMS |
|---|---:|---:|---:|
| free heat K versus NEMO `avt` | `5,721 / 17,400` | `1.594045423436441e-09` m2/s | `1.380750544672839e-11` m2/s |
| free complete K versus NEMO `zwt_mix` | `17,400 / 17,400` | `2.731312717075364e-03` m2/s | `1.270216333377450e-04` m2/s |
| heat-arm complete K | `17,400 / 17,400` | `2.731312717075364e-03` m2/s | `1.270216333966074e-04` m2/s |
| direct complete-K arm | `0 / 17,400` | `0` | `0` |

The non-vacuity plant advances recorded `zwt_mix[19,27,2]` by one binary64
ULP on step 1081 only.  It moves four registered matrix cells and 17,958
day-240 temperature cells, prints `STATUS PLANT-FIRED`, and exits 1.  Thus the
direct seam is observed at both its immediate consumer and the endpoint.

## Developed day-240 ranking

All arms start from NEMO's admitted step-1080 restart and advance the same 360
production-jitted steps to step 1440.  Values are wet-field RMS against NEMO's
step-1440 restart.

| arm | T (K) | S | U (m/s) | V (m/s) | SSH (m) |
|---|---:|---:|---:|---:|---:|
| free | `1.584259320940647e-02` | `6.925481217613344e-04` | `2.966753454580933e-04` | `3.741316307749545e-04` | `1.641114704244277e-04` |
| identity post-add | `1.584259320940647e-02` | `6.925481217613344e-04` | `2.966753454580933e-04` | `3.741316307749545e-04` | `1.641114704244277e-04` |
| NEMO heat K | `2.686974341191680e-03` | `6.147083204532816e-04` | `5.206152851742381e-05` | `1.579783491169916e-04` | `5.819685878568191e-05` |
| NEMO complete K | `1.241263037891706e-03` | `1.979791940553081e-04` | `6.361514024561804e-05` | `2.197654078310869e-04` | `5.821426550262672e-05` |

The direct arm removes `92.1650261324976%` of the free T RMS, versus
`83.03955478957635%` for heat alone, confirming the frozen ranking.  This is a
boundary sensitivity, not a candidate: replacing a downstream recorded array
is not itself a compiled NEMO statement.  Therefore no short ladder, month,
year-from-rest, DINO, generic-card, tank, or ORCA2 landing gate is applicable,
and no production row moves.

## First remaining non-bit operand

The current-tip one-step production walk reproduces Round 164's complete
vertical table exactly.  Once the direct arm makes `zwt_mix` BIT, the first
remaining operand in the executed source order is `e3w(Kmm)` in the divisions
on compiled lines 468-469, not `e3t(Kaa)` on the following line.

| source-order operand | unequal / scored | maximum absolute | RMS |
|---|---:|---:|---:|
| complete `zwt_mix` under direct arm | `0 / 17,400` | `0` | `0` |
| live `e3w(Kmm)` divisor | `17,400 / 17,400` | `1.964508555829525e-10` m | `4.058309524120698e-11` m |
| live `e3t(Kaa)` diagonal weight | `18,000 / 18,000` | `3.932996150979307e-10` m | `8.159729031748704e-11` m |
| temperature content RHS | `18,000 / 18,000` | `3.769802907231679e-02` K m | `7.046353169351283e-04` K m |

This names the next operand; it does not attribute the remaining `7.835%` of
day-240 error to thickness.  A day-240 one-variable substitution is still
required before that producer is a magnitude candidate.

## Review, citations, and tests

The required separate review command was run with `codex exec --sandbox
read-only`.  Its exact disposition is:

> independent review unavailable in-sandbox

The terminal error was `failed to initialize in-process app-server client:
Read-only file system`; it emitted no SHIP or DO NOT SHIP verdict.

The focused year-owner, compiled tracer-matrix, implicit-mixing, and literal
solver suites report `101 passed in 79.14s`.  The final citation-gate and full
tree results are recorded below after the receipt is mapped.

## Evidence hashes

| artifact | SHA-256 |
|---|---|
| `daily_record_audit.json` | `82e470886e0d8f10dacc21c25bb1d28390d67c62eda3b958d8113c3e0940b7e2` |
| `developed_vertical_sensitivity.json` | `ce7a243f34972f49d4634db15bdeeb8731029caee39cbd4a352a0184d23bd52d` |
| `vertical_walk.json` | `aba4fb219743035b71ae9ebf0eb5f4a92a416b2d4a0a87577a3ec17c4d217043` |
| `complete_K_ulp_plant.log` | `410fe15749f5da0c31f007f7f4f7ae2440d1cc40df5a34e19ea4df2f77237055` |
| `codex_review.log` | `eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5` |

## OPEN — round 169

1. Keep Round 163 as the immutable production arm.  Do not land the direct
   coefficient intervention; it is diagnostic-only.
2. Preregister a one-variable developed day-180-to-240 intervention replacing
   only the live `e3w(Kmm)` divisor with NEMO's recorded field at every step.
   Use the existing vertical record, a byte-exact identity arm, production JIT,
   and a consumed one-ULP plant.  Compare its removed day-240 T RMS with the
   complete-K arm's remaining `1.241263037891706e-03` K.
3. If `e3w(Kmm)` carries the remainder, walk its live free-surface/thickness
   producer to the first non-bit compiled statement.  If it does not, exonerate
   that operand by magnitude and rank `e3t(Kaa)` and the content RHS next.
4. Do not return to the TKE `rn2` rounding floor or the held shear route.  No
   NEMO acquisition, configuration decision, or carried-state change is
   requested.
