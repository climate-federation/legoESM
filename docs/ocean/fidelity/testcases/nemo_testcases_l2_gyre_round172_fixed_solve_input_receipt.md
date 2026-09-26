# Round 172 receipt — fixed solve inputs and paired NEMO record

**Status: STOPPED_FOR_RECORD.**  No production physics, configuration,
carried state, restart schema, card default, or immutable before arm changed.
The fixed-shape JAX identity prerequisite is **REFUTED**: the disabled contract
is byte-identical to the ordinary production step for all steps 1081--1440,
but selecting the model's own six live values moves `7,558` state bytes at
step 1081 (T `3,849` cells, S `3,680`, U/V/SSH zero).  Both directed JAX
sensitivities are withheld and no e3t/content owner is named.

The preregistered fallback is ready for operator acquisition.  An ordinary
production observer recorded 360 frames of `e3t(Kaa)` and temperature content
without moving one carried-state byte.  The new NEMO target runs independent
baseline, e3t-only, and content-only arms through step 1440; its source patch
passes `gfortran -fsyntax-only`, its 167,132,160-byte input layout passes, and
its truncation plant exits `1` with a named refusal.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round172/`.  The original
preregistration is commit `395da1b3b`; the post-refutation NEMO-pair
preregistration is `143dadecf`; the passive legoESM record is from clean
commit `010c1f459`.

The immutable production headline remains Round 163: first-over-bar kt3,
day-30 T RMS `6.572574374770603e-05` K, day-240
`1.644836070117868e-02` K, and day-360 `1.122566001855131e-02` K.  No accepted
trajectory candidate ran, so the ladder, month, year, DINO, generic GYRE,
tanks, and ORCA2 rows have zero registered movement.

## Compiled statements

The record build's executing adaptive-implicit branch forms the two
`e3w(Kmm)` off-diagonals and consumes `e3t(Kaa)` in the diagonal at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:468-474`.
It forms before content plus accumulated now tendency at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:549-567`, then
consumes the matrix and content in the backward recurrence at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:577-582`.

The proposed NEMO record changes only those two consumed inputs over steps
1081--1440: one arm substitutes the ordinary legoESM `e3t(Kaa)` at the matrix
statement, and the other substitutes ordinary legoESM temperature content at
the RHS statement.  Salinity content is never changed.  Because each NEMO arm
consumes the free legoESM sequence rather than its own evolving operand, the
result will rank family leverage only; it is not source-exact landing proof.

## Fixed-shape production result

The Round-132 daily record is admitted: all 360 daily boundaries exist and
all 12 overlaps with the monthly record are bit-identical.  The Round-123
process and Round-125 vertical admissions pass.  At step 1081, NEMO's literal
rebuild remains BIT for `zwt_mix`, `zwi`, `zwd`, `zws`, `zwt_lu`, `rhs_T`,
`fwd_T`, and `sol_T` (zero unequal cells for each).

| frozen prediction | verdict | result |
|---|---|---|
| disabled and identity contracts remain byte-identical | **REFUTED** | disabled passes all 360 steps; identity moves `7,558` bytes at step 1081 |
| free and complete-K/e3w reproduce Round 169 | **WITHHELD** | fail-closed stop before directed baselines |
| consumed e3t and content endpoints are BIT | **WITHHELD** | identity prerequisite failed |
| neither input carries half the `1.241262968697578e-03` K remainder | **UNMEASURED** | no controlled JAX trajectory |
| e3t removes more than content | **UNMEASURED** | no ranking exists |
| directed scale plants reach day 240 and exit nonzero | **WITHHELD** | plants are downstream of identity prerequisite |

This is a production-step result through `self._step_jitted`, not an isolated
closure.  The fixed contract keeps tuple shape constant and uses dynamic
selectors, but selecting identity values still changes XLA's fused graph.
That is precisely the eager-versus-production risk the identity prerequisite
was designed to detect.  The first evidence artifact used the imprecise
sentence “disabled or identity”; its detailed row shows only `fixed_identity`
failed.  The emitter is corrected prospectively without altering the frozen
artifact.

## Passive input record and acquisition controls

The legoESM exporter begins from the admitted NEMO step-1080 restart and
advances an independent ordinary production state.  Its observer records only
`e3t_Kaa.npy` and `content_T.npy`, each shape `[360,22,32,30]`, float64.  The
observer and ordinary carried states differ by `0` bytes over all 360 steps.
A `1 + 2**-20` first-frame scale control moves all 18,000 selected cells in
each family.

The acquisition script uses new target
`GYRE_OMIP_L2_P3_SM_R172SOLVEPAIR`, copies the Round-125 source card
file-by-file, applies an additive patch, and runs baseline/e3t/content from
rest to step 1440.  Admission requires baseline steps 1080 and 1440 to be
byte-identical to the producing Round-125 run and both directed step-1440
restarts to differ from baseline.  Every refusal is named; timing uses shell
`SECONDS`, never `/usr/bin/time`.

The first preflight correctly exposed an invalid compile-time use of runtime
`jpi`; it is preserved in `acquisition_preflight.log`.  The byte-position is
now computed at runtime with 64-bit arithmetic.  The corrected preflight
prints `SYNTAX_PROOF_PASS`, `SOLVE_INPUT_LAYOUT_PASS bytes=167132160`, and
`ROUND172_SOLVE_INPUT_PAIR_PREFLIGHT_READY`.  Removing eight bytes from the
raw input produces `REFUSE: solve-input raw size is 167132152, expected
167132160` and exits `1`.

## Landing verdict and scope

No first non-bit statement is named.  The cited matrix and RHS statements are
the two magnitude boundaries awaiting a controlled ranking.  The private
diagnostic contract and exporter do not alter any card's ordinary call; the
NEMO source patch is acquisition-only.  No Rule-12 or Decision-43/45/55/59
landing gate is applicable this round.

## Review, citations, and tests

Independent review, citation-gate, and focused-test results are recorded in
the final committed revision of this receipt.

## Evidence hashes

| artifact | SHA-256 |
|---|---|
| `daily_record_audit.json` | `9162f9e4526eaf29e1088406fedb612ef654165e1ae98176ed04375e51ab176d` |
| `developed_solve_input_sensitivity.json` | `7a417df1c92021d437c9398eb6a2e5cc685cec313304610fd3b18e8706985ae3` |
| `daily_record_audit_export.json` | `c88a9d49154c1305c664c28bba422b84f151fb19de032c28f710871e629e6d8f` |
| `lego_solve_inputs_report.json` | `c18dd2168cf69c033b0bdb65faa7617064521d26f1b4b9ec9bbb38402897d907` |
| `acquisition_preflight_fixed.log` | `a19547785d098d6cd88795c9a77fbc1b56e42d942f55c7ccb2de62b7949175b8` |
| `acquisition_truncation_plant.log` | `6595ee6362a8d94361cb65d47064b92d147283b9927b0f148a57bdff3b64d5aa` |

## OPEN — round 173

1. The operator runs the preregistered Round-172 acquisition script.  Do not
   rebuild if its admitted target and three records already exist.
2. Admit baseline only if steps 1080 and 1440 are byte-identical to the
   Round-125 source, and refuse either directed arm if it does not differ at
   step 1440.
3. Score baseline, e3t-only, and content-only day-240 restarts against NEMO.
   Rank the reduction from the production `1.241262968697578e-03` K
   complete-K/e3w remainder.  Promote the larger family to its producer walk;
   do not treat this forced-input ranking as a landing proof.
4. Keep Round 163 as the immutable production before arm.  No configuration
   or carried-state decision is requested.
