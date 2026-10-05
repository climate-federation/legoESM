# ORCA2 round 98 — rung-0 EEN Coriolis discriminator

Date: 2026-10-02. Base `96ded5336`; measurement and control tip
`2974562c4`. Scope is ocean only. Every number below is labelled
**independent**: hierarchy rung 0 starts from NEMO's own from-rest state.

## Verdict

**STOPPED_FOR_RECORD.** The source `e3f_0vor` divisor, its live free-surface
stretch, and the northern-fold association do not own round 97's remaining
Coriolis residual. The source and current live divisors are bit-identical in
all 799,200 values, including the fold row; consequently the fold-only arm
moves zero values in each of the eight frozen coefficients and leaves every
scored row unchanged.

The existing stream cannot close the next discriminator. Its historical
frozen-coefficient dump is rank-0-only, while the first nonzero residual's
maximum is at `(j=147, i=134)`, in rank 1's owned slab. A committed,
content-pinned acquisition therefore writes all eight frozen coefficients on
both ranks at `kt=1`, parses their self-describing headers, requires exact
rank coverage, and requires all twenty terminal restart shards to remain
byte-identical to round 96. Its syntax and layout preflight passes. The
operator must run it before the coefficient-construction statement can be
separated from the four-pair application.

No `packages/` file differs from the round base. No card field, configuration
choice, stabilizer, carried-state convention, threshold, sea-ice field, or
`unmeasured_features` entry changes. No physics statement lands.

## Source-ordered statement

NEMO first constructs the frozen EEN coefficient arrays in
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:1213-1265`, then consumes
them in the written four-U/four-V product and pairwise-sum order in
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:1369-1392`. For
`nn_e3f_typ=0`, the input divisor is the masked four-cell mean, F-fold
exchange, and dry-point replacement in
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynvor.f90:912-937`.

The round-97 baseline and source-divisor score dictionaries reproduce exactly
at this round's base, confirming R98-P1. Under the source divisor, the first
non-bit statement remains substep-1 `dyn_cor_2D`: 398 active U values and 400
active V values differ only by the sign bit of zero. The first nonzero row is
substep-2 U, with 68 differing active values and maximum absolute error
`2.9617669311254642e-8` at `(147,134)`.

## Registered arms

The committed CPU/fp64/JIT probe first proves that its separately built eight
coefficients are bit-identical to all eight coefficients exposed by the
production trace: zero unequal values and zero maximum error for every array.
It then applies the compiled product association with strict materialized
binary64 operations to the same recorded mid-step velocities.

| Boundary | U result | V result |
|---|---:|---:|
| substep 1, strict source association | 398 signed-zero differences | 397 signed-zero differences |
| substep 2, strict source association | 68 differences, max `2.9617669311254642e-8` | bit-exact |

The strict association corrects three V zero signs but does not close the
first non-bit row. R98-P2 is therefore **CONFIRMED** in its magnitude claim
and **REFUTED** as a complete construction owner: every substep-1 difference
still has zero magnitude, but the written association alone leaves 795
signed-zero differences.

The live-divisor arm is conclusive:

| Registered comparison | Unequal values | Maximum absolute error |
|---|---:|---:|
| full source live divisor vs current live divisor | 0 / 799,200 | 0 |
| source vs current outside northern fold | 0 | 0 |
| fold-only replacement vs current divisor | 0 / 799,200 | 0 |
| each fold-only frozen coefficient vs source coefficient | 0 / 26,640 | 0 |

R98-P3 is **REFUTED**: the northern-fold association moves neither the
`(147,134)` maximum nor any other coefficient. This is a measured
exoneration, not a landing. R98-P4 is not invoked because the final package
tree is identical to the base; GYRE is unchanged by construction.

The complete measurement is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round98/coriolis_residual_final.json`.

## Acquisition and controls

The operator launcher is
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round98_een_coeff_acquisition/run.sh`.
It creates the new target `ORCA2_OMIP_L4_R98EENCOEFF` and run directory
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round98/acquisition/orca2_rung0_een_coeff_ranked_10step_np2`.
It pins the source deck, binary, namelist, manifests, patch, writer, checker,
launcher, and preregistration by content rather than by a moving commit
allowlist. The patch is additions-only and the compiled preflight contains
exactly one executing dump call. Each record declares its magic, version,
step, time level, scheme, rank, local shape, global origin, owned bounds,
precision, field count, and each field's name/rank/dimensions/payload.

Preflight reports `SYNTAX_PROOF_PASS` and
`ORCA2_ROUND98_EEN_COEFF_PREFLIGHT_READY`. The layout and producer-content
plants both fire with exit 69. The record checker additionally carries header,
field-name, field-dimension, truncation, missing-field, all-zero payload,
swapped-rank, and restart-byte plants; they will run during admission because
no record exists yet.

The measurement's one-ULP application plant fires. Its first coefficient
plant was **RETRACTED** because the chosen coefficient multiplied an operand
whose output rounded back to the same bit pattern. The repaired control first
finds an active cell where one coefficient ULP reaches the measured Coriolis
output, then mutates only that cell; it fires at `ffu_nw[7,42]`. The failed
control remains documented rather than being counted as evidence.

## Tests, citation gate, and review

The focused round-95/97/98 record, walker, acquisition, and plant tests pass
22/22. The single `tests/ocean/fidelity -n 12` battery reached 98%, printed
seven failure markers, and then ended without a terminal summary or a live
pytest process. It is recorded as **incomplete**, not PASS. An allowed
isolation run confirms the same six established failures as round 97:

- `test_nemo_testcase_l2_gyre_round129_spread_floor_gate.py::test_record_backed_gate_passes`;
- `test_nemo_testcase_l2_gyre_round51_live_operands.py::test_live_trace_and_raw_history_arms_are_private_and_off_by_default`;
- `test_nemo_testcase_round35_stamp_scope.py::test_every_driver_that_arms_the_escape_scopes_it`;
- `test_recipe_case_board.py::test_every_oracle_comparison_has_a_row`;
- `test_nemo_testcase_worktree_stamp.py::test_every_report_emitter_stamps_the_worktree`;
- `test_nemo_si3_scalarmath_v2_gate.py::test_full_v2_gate_and_plants`.

The summary-less seventh marker cannot be assigned to a test ID; no failing
round-98 focused test exists. The wide-run log is retained in the external
round evidence rather than being promoted to a passing claim.

The default citation receipt passes with 274 citations, zero failures, and
zero unmapped citations. This receipt passes with three citations and the
same zero-failure result. A rigid +2 shift of the `dyn_cor_2D` application
citation is rejected by the plant.

The required separate `codex exec --sandbox read-only` review did not reach
the diff: `failed to initialize in-process app-server client: Read-only file
system`. Verdict: **independent review unavailable in-sandbox**.

## OPEN

1. The operator runs the round-98 launcher. Admit only two rank-complete,
   self-describing coefficient records plus 20/20 byte-identical terminal
   restarts, with every checker plant firing.
2. Assemble the NEMO coefficients over the owned domain and compare all eight
   arrays bit-for-bit with the production-JIT source-divisor coefficients.
   If the first mismatch is in coefficient construction, walk its compiled
   operand order. If all coefficients are exact, the signed-zero/product-sum
   application is the named owner.
3. After that discriminator, close the Coriolis row before advancing to the
   compiled drag and velocity-update statements.
4. When Decision 84's shared landing reaches this branch, remeasure rather
   than duplicating it. The rung-0 given-entry ladder, independent ladder, and
   independent month remain subsequent hierarchy deliverables.

## UNVERIFIED

- Rank-complete NEMO frozen coefficients do not yet exist; coefficient
  construction versus application remains unresolved.
- The first non-bit statement is still the substep-1 four-pair Coriolis
  application, not a landed statement.
- The rung-0 ten-step ladders and month have not yet been scored.
