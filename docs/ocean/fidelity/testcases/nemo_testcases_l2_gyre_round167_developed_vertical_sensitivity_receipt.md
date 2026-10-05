# Round 167 receipt — developed vertical-coefficient sensitivity

**Status: HELD.**  No production physics, configuration, carried state,
restart schema, or immutable before arm changed.  Replacing only the heat
diffusivity consumed by each developed step with NEMO's recorded `avt`
reduces the day-240 temperature RMS from
`1.584259320940647e-02` to `2.686974341191680e-03` K, a reduction of
`1.315561886821479e-02` K (`83.03955478957635%`).  This is a large bounded
TKE/closure-family sensitivity, but it does **not** complete the requested
TKE-versus-implicit-solve ranking: the preregistered complete-coefficient
preimage was REFUTED before its trajectory ran.  No first non-bit compiled
statement is named from an incomplete ranking.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round167/`.  The frozen
preregistration is commit `8d6a3a022`; the final scientific measurement is
`developed_vertical_sensitivity_v4.json` at clean instrument commit
`0736aca9a`.  The plant-only optimization is commit `bdd03b753`; it does not
change the ordinary scientific mode.  The immutable production headline
remains round 163: kt2 T/S/U/V AT-BAR, first-over-bar kt3, day-30 T RMS
`6.572574374770603e-05` K, day-240 `1.644836070117868e-02` K, and day-360
`1.122566001855131e-02` K.

## Compiled-source statements

The record-producing compiled program dispatches the selected closure, copies
its `avt_k` result into `avt`, and then calls EVD at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdfphy.f90:334-359`.
The TKE conversion from post-sweep energy and mixing length to the published
viscosity/diffusivity occupies
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdftke.f90:681-692`.
The active EVD replacement itself is
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdfevd.f90:108-109`.

The temperature operator forms the complete coefficient `zwt = avt +
ah_wslp2` and builds the tridiagonal matrix at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:418-480`; its three
solve recurrences are `:527-582`.  Therefore the two preregistered boundaries
are NEMO's published heat coefficient before the addition and the complete
coefficient after it.  These are measured source boundaries, not a new
configuration selector or stabilizer.

## Production-JIT intervention and controls

The existing year-owner instrument loads NEMO's admitted step-1080 restart
and independently advances steps 1081--1440 through
`LatLonCGridOceanModel.step -> self._step_jitted`.  At each step the private
test seam changes only the heat coefficient immediately before the production
implicit solve.  The model's viscosity, prognostic TKE update, forcing,
isoneutral coefficient, contents, geometry, and every carried state remain
the arm's own.  The Round-125 record is rebuilt from its own operands before
every coefficient is consumed; all matrix/solve calibration rows are BIT.

The free and identity arms complete all 360 steps with zero unequal state
bytes.  The first-step upstream process boundaries also remain unchanged in
each directed arm.  The frozen step-1081 calibration reproduces both older
measurements exactly:

| row | unequal / scored | maximum absolute | RMS |
|---|---:|---:|---:|
| vertical-diffusion T increment | `17,994 / 18,000` | `6.652851999202625e-04` K | `2.1834094362625713e-05` K |
| free heat K versus NEMO `avt` | `5,721 / 17,400` | `1.594045423436441e-09` m2/s | `1.380750544672839e-11` m2/s |
| free complete K versus NEMO `zwt_mix` | `17,400 / 17,400` | `2.731312717075364e-03` m2/s | `1.270216333377450e-04` m2/s |
| heat-arm complete K | `17,400 / 17,400` | `2.731312717075364e-03` m2/s | `1.270216333966074e-04` m2/s |

The heat arm's selected pre-isoneutral coefficient is BIT against recorded
NEMO `avt`.  It intentionally leaves the complete coefficient non-bit because
the model's own isoneutral operand remains in place.

## Frozen prediction refuted

The preregistration predicted that `NEMO(zwt_mix) -
legoESM(ah_wslp2)`, re-added by the production statement, would reproduce
NEMO's complete coefficient bit for bit.  It does not: `219 / 17,400`
interfaces differ, maximum `8.673617379884035e-19` m2/s, RMS
`7.888438065092211e-21` m2/s.  The prediction is **REFUTED** and retained.
The arm stops at step 1081 and its trajectory is not scored.  No adjacent-float
preimage, direct post-add seam, or tolerance was invented after observing the
failure.

Consequently frozen prediction 4—`effective_K` removes more day-240 error
than `heat_K`—is **UNMEASURED**, not confirmed or refuted.  The direct
post-add comparator is the missing discriminator.  Although its local
mismatch is at the rounding floor, transferring the heat-arm day-240 result
to that unrun trajectory would violate the campaign's one-variable rule.

## Registered developed day-240 rows

All values below are RMS against NEMO's admitted step-1440 restart.  The
identity row is an independent trajectory and equals free bit for bit.

| arm | T (K) | S | U (m/s) | V (m/s) | SSH (m) |
|---|---:|---:|---:|---:|---:|
| free | `1.584259320940647e-02` | `6.925481217613344e-04` | `2.966753454580933e-04` | `3.741316307749545e-04` | `1.641114704244277e-04` |
| identity | `1.584259320940647e-02` | `6.925481217613344e-04` | `2.966753454580933e-04` | `3.741316307749545e-04` | `1.641114704244277e-04` |
| NEMO heat K | `2.686974341191680e-03` | `6.147083204532816e-04` | `5.206152851742381e-05` | `1.579783491169916e-04` | `5.819685878568191e-05` |

The heat arm improves all five registered RMS rows.  Its T reduction is
`1.315561886821479e-02` K (`83.03955478957635%`).  For context only, the
committed Round-134 daily TKE-family reset removed
`1.210510385960158e-04` K (`0.736018%`), while the Round-126 inherited
vertical-process carry was `2.416827157805342e-02` K.  Those artifacts are
SHA-pinned in the JSON; they are not substituted for this arm.

This table is a developed-state sensitivity from NEMO's exact day-180 entry,
not a from-rest candidate and not a landing gate.  The short ladder, month,
year, DINO, generic recipe, tanks, and ORCA2 do not execute a production
change because no production change exists this round.

## Non-vacuity, review, citations, and tests

The record-backed plant advances one wet recorded `avt` value at
`[j,i,k] = [1,20,0]` by one binary64 ULP on step 1081 only.  It moves four
registered matrix cells and 17,959 day-240 temperature cells, prints
`STATUS PLANT-FIRED`, and exits 1.  The plant is therefore sensitive at both
the consumed solve and trajectory endpoint.

The required separate review command was run with `codex exec --sandbox
read-only`.  Its exact disposition is:

> independent review unavailable in-sandbox

The terminal error was `failed to initialize in-process app-server client:
Read-only file system`; it emitted no SHIP or DO NOT SHIP verdict.

The receipt citation gate reports PASS: five citations, zero failures, zero
unmapped citations, and zero map-audit failures.  Shifting the compiled
closure citation produces `SYMBOL-NOT-AT-LINE`, FAIL, and exits 1 as
required.

The focused owner/citation suites report `53 passed in 33.60s`.  The required
combined `tests/ocean/fidelity` plus `tests/ocean/unit` invocation was run once
with 12 CPU workers.  At 39%, one worker aborted inside JAX compilation and
was replaced.  At 48%, another worker hit `MemoryError` while pytest formatted
a failure, and xdist terminated with an internal error (exit 3).  Its partial
terminal summary is `18 failed, 4089 passed, 32 skipped, 2 xfailed, 12
warnings in 500.96s`.  The internal error emitted no `FAILED <node-id>` lines,
so those 18 partial failures cannot honestly be diffed against the recorded
87-node known-red set.  This is **INCOMPLETE TREE COVERAGE**, not a pass and
not evidence of a new round-167 failure; the two directly changed test files
are covered by the clean focused result.

## Evidence hashes

| artifact | SHA-256 |
|---|---|
| `developed_vertical_sensitivity_v4.json` | `80b70f54bed456ce5a0f3fafe63cbaec1fe3e4b05021b34fd2431bf4bd2f97f1` |
| `daily_record_audit_0736aca.json` | `081fabb44420c2e4444aab26d2c382713559f23c288566b2002ca01745b20eec` |
| `developed_vertical_avt_ulp_plant.log` | `c30bd2eb1a0ee4bdbd6bac175fbce96541676663b4fcebc72da0211365dc0b80` |
| `focused_tests.log` | `39b2f1854e9f358417553a39cc90446c107598097495b536f1274fbd03dce99f` |
| `full_fidelity_unit_tests.log` | `fef8c7e812ae17049c10eff57cba63882553cda362d813ebcea01a54d31c182b` |

## OPEN — round 168

1. Keep round 163 as the immutable before arm.  Do not land a heat-profile
   replacement: it is a family sensitivity, not a compiled NEMO statement.
2. Preregister a private **direct post-add complete-coefficient** intervention
   at the existing tracer-solve test boundary.  It must replace only the
   already-formed effective temperature coefficient with recorded NEMO
   `zwt_mix`, retain model viscosity and every other operand, pass a byte-exact
   identity arm, and use the same developed day-180-to-240 score.  This closes
   the comparator that subtraction/addition association made unmeasurable.
3. If direct complete K removes more error than heat K, move to the first
   non-bit implicit-solve operand.  If it does not, the heat/TKE family is the
   magnitude owner; then walk the post-sweep heat-coefficient program in
   compiled order and name the first non-bit statement that carries the
   measured day-240 response.  Do not return to the `rn2` rounding floor.
4. No NEMO acquisition, configuration decision, carried-state change, or
   pending decision 53/54/57/58 action is requested.
