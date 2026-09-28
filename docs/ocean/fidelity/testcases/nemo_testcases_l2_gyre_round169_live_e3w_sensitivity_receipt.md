# Round 169 receipt — live e3w(Kmm) sensitivity

**Status: HELD.**  No production physics, configuration, carried state,
restart schema, or immutable before arm changed.  After making NEMO's complete
tracer coefficient exact, replacing only the live `e3w(Kmm)` divisor reduces
the developed day-240 T3D RMS by `6.919412764115618e-11` K: only
`5.574493522233846e-08` of the remaining `1.241263037891706e-03` K.  The live
interface thickness is therefore exonerated by magnitude.  Round 170 must rank
the next two compiled-order boundaries, `e3t(Kaa)` and the tracer content RHS.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round169/`.  The original
preregistration is commit `4beac35aa`; the scientific instrument and run are
commit `86b962100`; the failed ULP control is registered at `f67914e1b`; and
the replacement control is commit `91191d9a4`.

The immutable production headline remains Round 163: kt2 T/S/U/V AT-BAR,
first-over-bar kt3, kt3 T RMS `8.60e-07` K, day-30 T RMS
`6.572574374770603e-05` K, day-240 `1.644836070117868e-02` K, and day-360
`1.122566001855131e-02` K.  This round runs no from-rest candidate, so none of
those production rows moves.

## Compiled statement and one-variable boundary

The record's compiled temperature branch first divides its already-formed
`zwt` coefficient by live `e3w(Kmm)` for the lower and upper coefficients at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:468-469`.  Only on
the next line does the diagonal consume `e3t(Kaa)`, at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:470`.  The resulting
matrix enters the ordered LU, forward, and backward recurrences at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:527-582`.

The existing private vertical-solve seam now accepts a fourth test array that
replaces only the tracer divisor.  Momentum face geometry, cell thickness,
content, coefficient, viscosity, closure state, forcing, and carried state are
unchanged.  Ordinary callers and the existing two- and three-array forms are
unchanged.  This is diagnostic scaffolding, not a configuration or candidate.

## Calibration and controls

The Round-132 daily record admits all 360 boundaries and all 12 monthly
overlaps at both clean instrument commits.  Every rebuilt Round-125 matrix and
solve field is BIT.  The free and complete-K arms reproduce Round 168 to every
printed digit.  Feeding the complete-K arm its own live divisor through the
new seam is byte-identical at every one of the 360 steps; the accumulated
mismatched-state byte count is zero.

At step 1081, the directed arm makes both consumed arrays BIT:

| row | unequal / scored | maximum absolute | RMS |
|---|---:|---:|---:|
| complete `zwt_mix` | `0 / 17,400` | `0` | `0` |
| live `e3w(Kmm)` | `0 / 17,400` | `0` | `0` |
| free heat K versus NEMO `avt` | `5,721 / 17,400` | `1.594045423436441e-09` m2/s | `1.380750544672839e-11` m2/s |
| free complete K versus NEMO | `17,400 / 17,400` | `2.731312717075364e-03` m2/s | `1.270216333377450e-04` m2/s |

All registered upstream process boundaries move by zero cells.

## Developed day-240 rows

All arms start from NEMO's admitted step-1080 state and advance the same 360
production-jitted steps.  Values are wet-field RMS against NEMO's step-1440
restart.  Every one of the 20 arm/field rows is registered.

| arm | T (K) | S | U (m/s) | V (m/s) | SSH (m) |
|---|---:|---:|---:|---:|---:|
| free | `1.584259320940647e-02` | `6.925481217613344e-04` | `2.966753454580933e-04` | `3.741316307749545e-04` | `1.641114704244277e-04` |
| NEMO complete K | `1.241263037891706e-03` | `1.979791940553081e-04` | `6.361514024561804e-05` | `2.197654078310869e-04` | `5.821426550262672e-05` |
| complete-K + divisor identity | `1.241263037891706e-03` | `1.979791940553081e-04` | `6.361514024561804e-05` | `2.197654078310869e-04` | `5.821426550262672e-05` |
| NEMO complete K + NEMO e3w | `1.241262968697578e-03` | `1.979791948696358e-04` | `6.361513938280375e-05` | `2.197654080968510e-04` | `5.821426937407114e-05` |

The directed divisor arm removes `6.919412764115618e-11` K, far below the
frozen half-remainder threshold `6.20631518945853e-04` K.  Prediction 4 is
CONFIRMED.  This sensitivity is `0.346` of the campaign's `2e-10` K
run-to-run floor in absolute T RMS and does not name a production fix.

## Plant retraction and replacement

Frozen prediction 5 is **REFUTED**.  Advancing one recorded divisor by one
binary64 ULP at step 1081 moved two matrix cells but zero day-240 temperature
cells.  The gate printed `STATUS PLANT-BLIND` and exited 2; the receipt does not
silently relabel that failed control.

The replacement was preregistered before implementation.  Scaling every
finite positive active divisor on step 1081 only by the exactly representable
`1 + 2**-20` moves all `17,400` targeted interfaces, `52,800` registered
matrix cells, and `17,973` wet day-240 temperature cells.  It prints
`STATUS PLANT-FIRED` and exits 1.  All steps after 1081 consume the unmodified
record.

## Scope, review, citations, and tests

No production statement changed, so the GYRE short ladder, month/year-from-rest
gate, DINO, generic GYRE recipe, LOCK_EXCHANGE, OVERFLOW, and ORCA2 do not
execute a candidate and have no moved row to register.  The Rule-12 and
Decision-43/45/55 landing gates are therefore not invoked.

The required separate review command was run with `codex exec --sandbox
read-only`.  Its exact disposition is:

> independent review unavailable in-sandbox

The terminal error was `failed to initialize in-process app-server client:
Read-only file system`; it emitted no SHIP or DO NOT SHIP verdict.

The final focused year-owner, literal-solver, and citation-map suite reports
`63 passed in 45.69s`.  The receipt citation gate reports PASS with three
citations, zero unmapped citations, zero failures, and zero map-audit failures.
Its shifted compiled-source plant reports FAIL and exits 1 as required.

The mandated single `tests/ocean/fidelity tests/ocean/unit -n 12` battery was
run once.  Eight JAX workers aborted during compilation and were replaced; the
run then remained at 95% without output for more than ten minutes and was
interrupted.  It emitted **no terminal pytest summary line** and no
`FAILED <node-id>` lines, so its interspersed failure/error counts cannot be
classified against the historical known-red set.  This is incomplete
full-tree coverage, not a pass.  The directly changed owner, literal-solver,
and citation files are covered by the clean 63-test focused result and were not
rerun after the one mandated full-tree attempt.

## Evidence hashes

| artifact | SHA-256 |
|---|---|
| `daily_record_audit.json` | `73b9171c2561650bb20ca4bdcb37494c66edb412c2b76c70c5a4abbba3a54b29` |
| `developed_e3w_sensitivity.json` | `a59b273ed814d24a150eea83c2b59e944d5fd66fd1a7c9d347a9353fac0311e9` |
| `e3w_ulp_plant.log` | `be9ea1e7333a70286b52c69045df1cf716a498d64729590af2e7030bbcbd1105` |
| `e3w_scale_plant.log` | `e6efe963c4fdc2069499e2cbc4873449f48afd0947b328508fb6bd3ac7999c51` |
| `codex_review.log` | `eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5` |
| `citation_gate.log` | `6bd5651808e081babb67e0df996a40441a5ddd2ca97f1b280df22ca5cbf3a4d1` |
| `citation_gate_shifted_plant.log` | `8794b6927191adfb00424d73430a8ef479ef1befe1eaf68cf0c25c4847f56c71` |
| `focused_tests.log` | `69f609030dedc86f711332c0b9e54b1ed73f6858cc6552b44868b68e95e9dff7` |
| `full_fidelity_unit_tests.log` | `78c5494986efeb1d084113ff264ea7ae5c188955259fc3393427b6f40ca6586a` |

## OPEN — round 170

1. Keep Round 163 as the immutable production arm.  Do not land the complete-K
   or live-divisor substitutions; both are diagnostic-only.
2. Preregister two developed day-180-to-240 arms at the existing solve seam:
   after exact complete K and exact `e3w(Kmm)`, replace only `e3t(Kaa)`, then
   replace only the already-formed tracer content RHS.  Each needs its own
   identity arm, production-JIT score, and consumed endpoint plant.
3. Rank the two arms by removed day-240 T RMS.  Walk the winning producer to
   its first non-bit compiled statement.  Do not return to TKE `rn2`, the held
   shear route, or the exonerated live `e3w(Kmm)` divisor.
4. No acquisition, configuration decision, or carried-state change is
   requested.
