# ORCA2 round 106 — EEN accumulator/scale discrimination

Date: 2026-10-02. Base `edce5d5070a955c2508d7258aeacdb1994980be9`.
Scope is ocean-only measurement on hierarchy rung 0. Every numerical result
below is **independent**: the rung starts from NEMO's own from-rest state.

## Verdict

**HELD.** The existing round-105 acquisition is admitted without rerunning
NEMO. It proves that the EEN coefficient accumulator is the first recorded
non-bit boundary, and it exposes a genuine accumulator/final-scale cancelling
pair at the northern-fold `ffv_nw` and `ffv_ne` coefficients. The tempting
scale-only production fix was falsified and reverted. No model physics, card
field, configuration value, threshold, stabilizer, carried state, sea-ice
selector, or ORCA2 `unmeasured_features` entry changes in the final tree.

## Existing-record admission

The round-105 launcher refused only because it expected a nonexistent
`ocean.output_0001`. NEMO wrote rank 0 recorder markers to `ocean.output` and
rank 1 markers to `run.user.stdout.log`. The repaired admission checks those
actual files and requires one initialization plus one dump for each distinct
rank. Its rank-log plant removes that distinction and fires.

The existing target then passes every substantive gate:

- both self-describing records parse with exactly-once global rank coverage;
  their SHA256 digests begin `afb887211a66` and `50d8c51f16bf`;
- all eight final coefficient arrays reconstruct bit-for-bit as recorded
  accumulator times recorded scale;
- all twenty kt=1..10 terminal restart shards are byte-identical to the
  admitted round-98 reference; and
- the eight inherited payload/restart plants and the new rank-log plant fire.

The admission ends with `STATUS PASS_R104_EEN_ACCUM_ADMISSION`. No new NEMO
acquisition is needed.

## Source order and the first recorded non-bit boundary

The compiled record producer forms the U-side `zpvo_nw` triad and then updates
`ffu_nw` in source order
(`ORCA2_OMIP_L4_R105EENACC/BLD/ppsrc/nemo/dynspg_ts.f90:1231-1244`). Direct
instrumentation of legoESM's live builder shows that the emitted `ffu_nw`
accumulator is already non-bit: 3,953 bit differences, including 438 magnitude
differences (437 off the northern fold). This is the first newly measured
non-bit boundary; the next walk must split the cited recurrence's inputs before
assigning ownership to one RHS term.

The compiled V-side recurrence is at
(`ORCA2_OMIP_L4_R105EENACC/BLD/ppsrc/nemo/dynspg_ts.f90:1277-1278`) and the
subsequent final scales are at
(`ORCA2_OMIP_L4_R105EENACC/BLD/ppsrc/nemo/dynspg_ts.f90:1288-1289`). These are
the corrected producer lines; the preregistration keeps and loudly corrects
its earlier diagnostic-layout line numbers.

## Accumulator/scale discrimination

All eight current scales are bit-exact off the fold. The current U scales are
bit-exact everywhere; only `scl_v_nw` and `scl_v_ne` differ, each at 68
northern-fold cells. The non-fold final coefficient differences are therefore
accumulator-owned signed zeros: 3,514/3,515/3,574/3,577 for
`ffu_ne/nw/se/sw`, and 3,505/3,504/3,585/3,584 for
`ffv_ne/nw/se/sw`. Substituting the recorded accumulator makes substep-1 U and
V bit-exact without moving coefficient magnitudes.

The direct live trace records accumulator magnitude differences in every
coefficient:

| coefficient | bit unequal | magnitude unequal | non-fold magnitude | fold magnitude |
|---|---:|---:|---:|---:|
| `ffu_ne` | 3,940 | 426 | 425 | 1 |
| `ffu_nw` | 3,953 | 438 | 437 | 1 |
| `ffu_se` | 4,009 | 438 | 437 | 1 |
| `ffu_sw` | 4,013 | 441 | 440 | 1 |
| `ffv_ne` | 3,990 | 486 | 419 | 67 |
| `ffv_nw` | 3,987 | 483 | 416 | 67 |
| `ffv_se` | 3,933 | 348 | 347 | 1 |
| `ffv_sw` | 3,920 | 336 | 335 | 1 |

The northern V pair is decisive. For `ffv_nw`, replacing only the accumulator
leaves 66 magnitude differences and replacing only the scale leaves 66;
replacing both leaves zero bit differences. The corresponding counts for
`ffv_ne` are 67, 67, and zero. Accumulation and final scale are therefore a
measured cancelling pair, not two independently landable fixes.

With every recorded final coefficient substituted, substep 1 U/V is
bit-exact. Substep 2 V is active-bit-exact (68 full-domain signed-zero
differences only), while substep 2 U retains exactly 68 active unequal cells,
maximum `2.9617669311254642e-8 m s-2` at `(j,i)=(147,134)`. That later
consumer residual is unchanged and remains separate from this coefficient
construction walk.

## Frozen prediction disposition

- R106-P1, P2, P3, P5, P8, P10, and P11: **CONFIRMED**.
- R106-P4: **REFUTED**. The 66/67 fold magnitudes do move under accumulator
  substitution.
- R106-P6 and P7: **REFUTED**. The production scale-only arm leaves the same
  66/67 magnitude differences. The apparent offline closure had changed both
  accumulator and scale, violating the one-variable rule.
- R106-P9: **REFUTED AS WRITTEN**. The predicted fold accumulator differences
  exist, but 335–440 non-fold magnitude differences also exist.

The rejected scale arm is preserved in history as `8bad3da54` and explicitly
reverted by `b53fd55de`. The direct accumulator trace hook is likewise
preserved as measurement instrumentation and reverted. The final production
tree has no `packages/` difference from the round base.

## Mechanical evidence and validation

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round106/`:
`admission.log`, `een_operand_discrimination.json`,
`een_fold_scale_arm.json`, `een_direct_accumulator.json`, and both firing
operand plants. The rank-log plant is in the same directory. The focused
record/probe battery passes 3/3.

The one prescribed `tests/ocean/fidelity -n 12` invocation reached 97% and
then stopped producing output after every pytest worker had exited. It was
terminated and is **not** counted as a pass: 2,250 tests reported PASS and six
reported FAIL before the missing terminal summary. The six are the known SI3
scalar-math, live-operands field-order, dirty escape-scope, worktree-stamp,
missing `hires_lane_surface` case-board row, and round-129 certified-year
harness stamp ratchets; none is in a file changed by this round.

Separate read-only Codex review was attempted. Verdict: **independent review
unavailable in-sandbox** (`failed to initialize in-process app-server client:
Read-only file system`).

## OPEN

1. Split the cited `ffu_nw` recurrence into its `zpvo_nw`, thickness, mask,
   and carried-state operands; name the first unequal input without combining
   it with the fold scale.
2. After that source-order walk, revisit the measured northern V
   accumulator/scale pair as a pair, with a known-answer control.
3. Keep the later 68-cell substep-2 U residual separate until coefficient
   construction is closed.
4. The package-exposed rung-0 card and independent 240-step month remain open
   hierarchy deliverables.

## UNVERIFIED

- Which individual RHS operand first makes the U accumulator non-bit.
- Whether one source-exact paired transcription can close the northern V
  coefficient pair without moving an earlier certified row.
- The owner of the later 68-cell substep-2 U residual.

## Choices

ASKED: Decisions 52, 80, 83, and 84 remain unchanged. UNASKED: none.
