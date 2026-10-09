# ORCA2 round 108 — EEN recorder-chain repair

Date: 2026-10-02. Base `18b659b033b1032bb9f4c29b3a6f244d16cae17f`.
Scope is instrumentation only on hierarchy rung 0. Any later record number is
**independent** because rung 0 starts from NEMO's own from-rest state.

## Verdict

**STOPPED_FOR_RECORD.** The operator's round-107 acquisition built the intended
binary but NEMO stopped during initialization because that binary retained the
round-105 accumulator recorder and the launcher did not supply its output
directory. A fresh-target, launcher-only resume now supplies both recorder
directories and is preflight-ready. No NEMO source, model physics, card field,
configuration value, threshold, stabilizer, carried state, sea-ice selector,
or ORCA2 `unmeasured_features` entry changes.

## Source-resolved failure

The failed run contains two zero-byte round-107 files and no operand payload.
Its `ocean.output` first reports `round105: missing EEN operand output
directory`, then `round105: cannot initialize EEN operand record`; the wrapper
therefore returns `STOP 123`. No physics number is inferred from that exit.

The exact compiled round-107 binary calls the retained round-105 initializer
before the new round-107 initializer
(`ORCA2_OMIP_L4_R107EENSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:1103-1104`). The
retained initializer reads `ORCA2_R105_EEN_ACCUM_DIR`, requires an absolute
path, opens a rank-tagged stream with `STATUS='NEW'`, and only then reports its
initialization marker
(`ORCA2_OMIP_L4_R107EENSTEP/BLD/ppsrc/nemo/l4_r105_een_accum.f90:34-45`). The
round-107 launcher exported only the new recorder's variable. This directly
explains all three error messages and the two zero-byte new streams.

## Repair and acquisition contract

The repair reuses the exact compiled binary (`3557a0c36665...`) and compiled
`dynspg_ts` (`ccb5b5523567...`) from the failed target. It creates a new run
directory under `round108/acquisition`, stages the checksum-pinned rung-0 deck
and inputs, and exports both recorder variables to that one absolute,
pre-created directory. The failed round-107 directory is retained unchanged.

Admission requires:

1. `STOP 0` and exactly one initialization plus one dump marker for ranks 0
   and 1 from both recorders;
2. two nonempty self-describing round-107 streams;
3. both inherited round-105 streams byte-identical to the admitted round-105
   record; and
4. all twenty kt=1..10 terminal restart shards byte-identical to the admitted
   rung-0 source run.

The checker parses names and dimensions from each self-describing field header.
Its header, field-name, field-dimension, truncation, missing-field,
duplicate-rank, bottom-index, recurrence, and restart-byte plants remain the
admission controls.

## Frozen prediction disposition

- R108-P1 through R108-P5 are **UNMEASURED** until the operator runs the fresh
  target. This round does not convert preflight success into runtime evidence.
- The earlier round-107 runtime expectation is **REFUTED**: the new writer was
  not the only live recorder in the inherited binary.
- Round 107's offline literal-loop result remains unchanged; its missing
  per-level oracle comparison remains unmeasured.

## Mechanical evidence

The launcher pins the failed binary, compiled source, namelist, deck manifest,
input manifest, checker, and preregistration by content. Its inherited-variable,
rank-log, and toolchain plants all fire. Shell syntax passes, the focused
launcher/checker plus citation tests pass 21/21, and the clean-tree preflight
ends:

```text
ORCA2_ROUND108_EEN_STEP_RESUME_READY /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round108/acquisition/orca2_rung0_een_step_ranked_resume_10step_np2
```

This round changes no `packages/` file. The GYRE, DINO, tank, rung-0 trajectory,
and rung-7 trajectory implementations therefore do not move.

The default citation gate and this receipt's gate both pass with zero unmapped
citations, citation failures, or map-audit failures. Shifting the compiled
initializer-call citation by two lines fires `SYMBOL-NOT-AT-LINE`.

The one prescribed `tests/ocean/fidelity -n 12` invocation reached 99%, with
2,286 passes and the six known failures (SI3 scalar math, live-operands field
order, dirty escape scope, worktree stamp, missing `hires_lane_surface`
case-board row, and the round-129 certified-year harness stamp), then stopped
in the pre-existing planted-control tail without a terminal pytest summary.
It was terminated and is not counted as a pass. No failure is in a file changed
by this round.

Separate read-only Codex review was attempted. Verdict: **independent review
unavailable in-sandbox** (`failed to initialize in-process app-server client:
Read-only file system`).

## OPEN

1. Operator runs the reported round-108 launcher; admission must satisfy
   R108-P1 through R108-P5 before any operand is quoted.
2. Once admitted, compare `zpvo_nw`, live U thickness, live V thickness,
   neighboring mask, stored product, and carried accumulator in compiled
   source order; name the first remaining signed-zero statement.
3. Only after `ffu_nw` is bit-exact, walk the separate 68-cell south-U debt,
   the northern V cancelling pair, and the later 68-cell substep-2 residual.
4. The package-exposed rung-0 card and independent 240-step month remain open
   hierarchy deliverables.

## UNVERIFIED

- Runtime success, restart identity, and observational passivity of the fresh
  acquisition until the operator runs it.
- Which recorded per-level operand first owns the 3,577 signed-zero differences.

## Choices

ASKED: Decisions 52, 80, 83, and 84 remain unchanged. UNASKED: none.
