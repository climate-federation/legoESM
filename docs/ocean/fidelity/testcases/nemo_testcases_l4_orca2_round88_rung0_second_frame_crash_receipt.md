# ORCA2 round 88 — rung-0 second frame crash

**Verdict: STOPPED_FOR_RECORD.** The operator-run round-87 optimized target
crashed at kt=1 after emitting only the two rank-local stage-0 frames. The
record is not admitted. A fresh exact-source debug acquisition is preflight
ready; no legoESM package, card, hierarchy switch, NEMO physics statement,
threshold, stabilizer, carried state, or sea-ice selector changed. Every
scientific label remains **independent**.

Base: `5317f16b452224049844fdf282dd75fd60daf738`. Preregistration:
`PREREG_nemo_testcases_l4_orca2_round88.md` was committed before the target was
measured.

## Record refusal and what remains valid

The target contains 2/80 entry/stage frames and 2/20 restart shards. Its MPI
log ends with signal 11 before any `STOP 0`. Therefore the round-87 admission
prediction is **REFUTED**, restart identity is **UNMEASURED**, and no stage-1
state or first differing stage boundary is available.

The isolated surface record itself parses to physical EOF with exactly 25
PRESENT and 10 ABSENT fields. That confirms only the round-87 surface schema;
it does not admit the failed trajectory. The compiled driver writes the
rank-local stage-0 frame before calendar and surface work
(`ORCA2_OMIP_L4_R87FRAMES/BLD/ppsrc/nemo/stprk3.f90:90-108`), so the two
available files are the true Nbb step-entry boundary. `output.init` remains
retracted as an entry operand.

## Retraction: the first repair was premature

The optimized addresses mechanically resolve the failure stack to
`tra_adv_trp_t` called from `stp_RK3_stg`; they do not carry source-line debug
information. The inherited stage-1 WZV probe does include an unconditional
canonicalization of `rnf`
(`ORCA2_OMIP_L4_R87FRAMES/BLD/ppsrc/nemo/traadv.f90:273-287`), while the
resolved rung-0 deck has `ln_rnf=F` and the compiled allocator allocates `rnf`
only inside that switch
(`ORCA2_OMIP_L4_R87FRAMES/BLD/ppsrc/nemo/sbc_oce.f90:195-216`). That makes the
runoff operand a candidate, not a proved faulting line: the truncated stream
ends near the preceding metric operand and compiler evaluation/buffering does
not establish which source item faulted.

An additions-only runoff guard was briefly committed as an acquisition
candidate, then retracted before operator use. Its launcher now exits 79 with a
named REFUSE before reaching any build or run action. No number or physics
claim depends on that patch.

## Exact debug acquisition

The replacement launcher clones the exact `ORCA2_OMIP_L4_R87FRAMES` source,
pins the failed binary and source hashes, and rebuilds under
`-O0 -g -fbacktrace -fcheck=bounds` with scalar math. It accepts only a nonzero
run whose backtrace names both `tra_adv_trp[_t]` and a concrete compiled
`traadv.f90:N` line. Debug values are diagnostic-only and cannot enter a
scientific gate.

Clean-tree preflight prints:

```text
ORCA2_ROUND88_RUNG0_FRAME_DEBUG_PREFLIGHT_READY /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round88/acquisition/orca2_rung0_frame_debug2_np2
```

## Frozen prediction ledger

| Prediction | Verdict |
|---|---|
| existing record admits 80 frames, 20 exact restarts, and every plant | **REFUTED — 2 frames, 2 restarts, SIGSEGV** |
| stage 0 is the true Nbb step-entry boundary | **CONFIRMED from the compiled call site and two rank-tagged frames** |
| at least one field changes from stage 0 to stage 1 at kt=1 | **UNMEASURED — no stage-1 frame exists** |
| frame comparison alone names a boundary but not one arithmetic statement | **UNMEASURED — comparison cannot start** |
| round-87 surface repair emits 25 PRESENT and 10 ABSENT entries | **CONFIRMED for the isolated partial record only** |

## Gates, tests, and review

The debug launcher passes `bash -n` and its clean committed-tree preflight.
The focused acquisition controls pass **3/3**. They require the speculative
launcher to refuse before its candidate, apply the additions-only patch at
fuzz zero, and pin the debug launcher to a fresh target with symbol and bounds
checks. The surface parser independently reports `PASS_SURFACE_ABSENCE`.

The required separate `codex exec --sandbox read-only` review did not read the
diff: `failed to initialize in-process app-server client: Read-only file
system`. Verdict: **independent review unavailable in-sandbox**.

No `packages/` file changed, so GYRE, DINO, tanks, the shipped ORCA2 card, and
its `unmeasured_features` ice tuple are unchanged by construction. No
configuration choice was made.

## OPEN

1. Operator runs the round-88 debug launcher and returns the source-resolved
   `traadv.f90:N` backtrace.
2. Repair only that proved write-only instrumentation statement, under a new
   optimized target and the existing 80-frame / 20-restart identity gate.
3. After admission, build the explicit rung-0 card, compare its entry with the
   stage-0 frames, and walk the first non-bit boundary. Rung 1 remains untouched.

Acquisition launcher:
`/tmp/autopilot-orca2-2942529579/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round88_rung0_frame_debug/run.sh`.

## UNVERIFIED

- The exact faulting `traadv.f90` line is not source-resolved yet.
- Stage 1 through stage 3, restart identity, rung-0 card ladders, and month are
  unmeasured.
