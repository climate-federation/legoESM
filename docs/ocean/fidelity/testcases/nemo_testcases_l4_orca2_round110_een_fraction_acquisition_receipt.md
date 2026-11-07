# ORCA2 round 110 — EEN fraction and southern-halo acquisition

Date: 2026-10-02. Base `e78352254c58083ed39409cbde81725cd18fa204`.
Scope is ocean-only instrumentation on hierarchy rung 0. Every eventual ocean
number is **independent** because rung 0 starts from NEMO's own from-rest
state.

## Verdict

**STOPPED_FOR_RECORD.** Round 109's first non-bit boundary remains NEMO's
three-term `zpvo_nw` assignment: 180 magnitude differences, all on global row
`j=0`, level `k=0`. The existing admitted stream contains only the completed
sum, so this round does not guess which quotient or halo operand owns it.

A committed, additions-only acquisition now records the west, center, and
south fractions plus every numerator and denominator component on both ranks.
Its clean-tree preflight and five launcher controls pass. The sandbox does not
run MPI under the standing operator rule, so none of R110-P1 through R110-P4
is measured and no physics statement lands.

The operator action is:

```text
/tmp/autopilot-orca2-1579113232/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round110_een_fraction_acquisition/run.sh --run
```

It creates the new target `ORCA2_OMIP_L4_R110EENFRAC` and the new run directory
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round110/acquisition/orca2_rung0_een_fraction_ranked_10step_np2`.

## Source-ordered discriminator

The compiled rung-0 U loop evaluates, in order, the west quotient, center
quotient, and south-halo quotient, then adds them into `zpvo_nw`
(`ORCA2_OMIP_L4_R107EENSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:1241-1245`).
The admitted round-109 recorder begins only after that completed assignment
and then records the downstream live thickness, mask, product, and accumulator
state (`ORCA2_OMIP_L4_R107EENSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:1256-1263`).
Therefore a new source-side record is required before the walk can advance.

The new writer stores these self-described fields for every executed U level:

- `west`, `center`, and `south` copies of `ff_f`, `e3f_0vor`, `r3f`,
  `fe3mask`, their parenthesized denominator, and their quotient;
- the west-plus-center partial sum and the final ordered sum; and
- `mbku`, with all arrays zero-initialized so writes outside the literal loop
  are mechanically rejected.

The instrumentation does not rewrite the oracle expression. One additional
call records the same operands after NEMO has evaluated its original
assignment; the admitted round-107 `zpvo_nw` remains the binding final-sum
reference. This keeps the patch additions-only while making a recorder replay
failure a hard refusal rather than evidence about physics.

## Admission contract and controls

The launcher pins the source `dynspg_ts`, CPP keys, binary, namelist, deck and
input manifests, and all five producer artifacts by SHA-256 content. It does
not pin a moving producer commit. It builds a fresh config and run directory,
uses one absolute pre-created per-rank output stream, and exports the retained
round-105 and round-107 recorder paths as well as the new round-110 path.

Admission requires all of the following:

1. `STOP 0`, exactly one initialization and dump marker for ranks 0 and 1 for
   all three live recorders, and two complete new self-describing streams;
2. inherited round-105 and round-107 streams byte-identical to their admitted
   round-108 counterparts;
3. every kt=1..10 restart shard byte-identical to the admitted round-108 run;
4. exact rank-complete owned-domain coverage, valid integral `mbku`, the NEMO
   dummy level untouched, and zeros outside every executed loop; and
5. bit-exact replay of all three denominators, quotients, the partial sum, the
   final sum, and the final sum against admitted `zpvo_nw`.

The header, field-name, field-dimension, truncation, missing-field,
duplicate-rank, bottom-index, denominator, quotient, sum, inherited-record,
and restart-byte plants are wired into admission. They cannot run until a
record exists. The five record-independent launcher controls all fire now:
layout, absolute path, three-recorder environment, producer-content manifest,
and rank-log completeness.

The clean committed preflight ends:

```text
SYNTAX_PROOF_PASS l4_r110_een_fraction.f90 dynspg_ts.f90
ORCA2_ROUND110_EEN_FRACTION_PREFLIGHT_READY /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round110/acquisition/orca2_rung0_een_fraction_ranked_10step_np2
```

Preflight and plant logs are under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round110/`.

## Prediction disposition

- R110-P1: **UNMEASURED** until restart and inherited-stream identity admit.
- R110-P2: **UNMEASURED**; west/center/south fraction counts require the new
  record.
- R110-P3: **UNMEASURED**; no southern-halo operand is named from an offline
  reconstruction.
- R110-P4: **UNMEASURED**; the new record must exclude a rank-seam owner.
- R110-P5: **CONFIRMED procedurally**; this round makes no product or
  accumulator signed-zero attribution while the earlier `zpvo_nw` boundary is
  open.

## Validation and review

The writer and patched compiled source pass an fp64 syntax compilation against
the exact round-107 build includes. The focused parser/schema plus citation
battery passes 20/20. The default cumulative citation gate passes 274
citations and this receipt passes two, with zero failures and zero unmapped
citations; shifting the first compiled range by two lines fires
`SYMBOL-NOT-AT-LINE`. No `packages/` file changes, so the GYRE, DINO, tank,
rung-0 trajectory, and rung-7 trajectory implementations do not move.

The one required `tests/ocean/fidelity -n 12` invocation collected 2,324
tests, reached 97%, and emitted the same six registered failure markers as
round 109 before the documented xdist-controller stall. It was interrupted
after a full minute with no output and is **incomplete, not PASS**. The six
markers are SI3 scalar math, live-operands private/default-off, dirty escape
scope, worktree stamp, missing `hires_lane_surface` case-board row, and the
round-129 certified-year record stamp. No round-110 test failed.

Separate `codex exec --sandbox read-only` review was attempted. Verdict:
**independent review unavailable in-sandbox** (`failed to initialize
in-process app-server client: Read-only file system`).

## OPEN

1. Operator runs the committed round-110 launcher. Admission must prove
   restart identity, inherited-stream identity, arithmetic replay, and every
   record plant before a fraction is quoted.
2. After admission, compare west, center, and south fractions in compiled
   source order, then the first unequal fraction's `ff_f`, `e3f_0vor`, `r3f`,
   and `fe3mask` operands. Preserve any failed frozen prediction as REFUTED.
3. Only after `zpvo_nw` is bit-exact may the walk return to the two product
   signed zeros, accumulator-addition signs, the separate 68-cell south-U
   debt, the northern V cancelling pair, or the later 68-cell substep-2 U
   residual.
4. The package-exposed rung-0 card and independent 240-step month remain open
   hierarchy deliverables.

## UNVERIFIED

- Which fraction or southern-halo operand owns the 180 row-0 magnitude
  differences.
- Runtime observational passivity of the round-110 recorder until the
  operator completes the acquisition.

## Choices

ASKED: Decisions 52, 80, 83, and 84 remain unchanged. UNASKED: none.
