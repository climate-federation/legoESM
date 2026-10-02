# ORCA2 round 90 — rung-0 runoff-probe repair

**Verdict: STOPPED_FOR_RECORD.** The source-resolved debug run proves that the
round-87 frame acquisition crashes in inherited write-only instrumentation,
not NEMO physics: the stage-1 WZV probe dereferences the unallocated runoff
array while rung 0 has runoff disabled. The exact previously retracted guard
is now justified, committed, and preflight-clean under a fresh optimized
target. No legoESM package, card, hierarchy switch, NEMO physics statement,
threshold, stabilizer, carried state, or sea-ice selector changed. Every label
is **independent**.

Base: `e454c8255`. Preregistration commit: `4aa514d19`. Repair commit:
`ae928e2f8`. The preregistration preceded inspection of the full operator log,
target run directory, run stdout, and compiled debug sources.

## Source-resolved fault

The round-89 debug executable ran once. Its target-local stdout survived the
launcher's nested ERR traps and records one rank-0 signal-11 exit. The two
launcher REFUSE lines are propagation of that one failure at the MPI command
and enclosing subshell, not two NEMO runs. The launcher stopped before writing
`run.exit_code.txt`; that is bookkeeping debt only, because the target-local
stdout and debug binary are both SHA-pinned by the replacement launcher.

NEMO's stage program calls tracer transport at
`ORCA2_OMIP_L4_R88FRAMEDEBUG/BLD/ppsrc/nemo/stprk3_stg.f90:555`. Inside the
inherited Lane-4 recorder,
`ORCA2_OMIP_L4_R88FRAMEDEBUG/BLD/ppsrc/nemo/traadv.f90:287` passes `rnf` to the
canonicalizer; the backtrace names its dereference at `:357`. The allocator proves ownership:
`ORCA2_OMIP_L4_R88FRAMEDEBUG/BLD/ppsrc/nemo/sbc_oce.f90:201` allocates `rnf`
only under `ln_rnf`, while the pinned rung-0 output resolves that switch false.
Thus R90-P2 and R90-P3 are **CONFIRMED** and round 88's deliberately retracted
candidate is now source-proved.

## Repair and acquisition bar

The repair adds three `IF( ln_rnf )` guards around all five writes and the
close on legacy unit 993. It removes no source line. The intervening `CALL wzv`
and all tracer-transport statements remain outside those guards, so the repair
cannot suppress physics. With runoff off the obsolete fixed-schema file is
absent rather than filled with plausible zeros; the separate surface-input
record remains self-describing and requires its four runoff fields to be
explicit ABSENT entries.

The fresh launcher pins the failed optimized record, the source-resolved debug
stdout and binary, the exact source hashes, the admitted rung-0 deck/input
manifests, and the round-83 calibration. Admission requires 80 rank-tagged
kt=1..10 entry/stage frames, all frame and surface plants firing, no legacy
stage-1 WZV file, 20 terminal restart shards, and byte identity for every shard
against round 83. Debug values are diagnostic-only.

## Frozen prediction ledger

| Prediction | Verdict |
|---|---|
| R90-P1: one debug run propagated through nested traps | **CONFIRMED** — one run stdout, one backtrace, one rank-0 signal exit, no second run target |
| R90-P2: target stdout contains a source-resolved line | **CONFIRMED** — the recorder call and dereference are the two source lines cited above |
| R90-P3: fault lies in additions-only recorder code | **CONFIRMED** — unconditional recorder access to an owner-disabled allocation |
| R90-P4: one minimal repair is enough to request the optimized record | **CONFIRMED for preflight; trajectory result UNMEASURED** — exact three-guard repair parses and compiles, record remains operator action |

## Gates, tests, and review

- Clean committed-tree preflight prints
  `ORCA2_ROUND90_RUNG0_FRAMES_PREFLIGHT_READY`; its missing-guard plant fires.
- Focused round-90 controls: **4 passed**; `bash -n` and `git diff --check`
  pass.
- The receipt citation gate passes this receipt and the default cumulative
  receipt with zero unmapped citations; shifting the dereference citation by
  two lines makes the gate fail.
- Required separate read-only review verdict: **independent review unavailable
  in-sandbox**. `codex exec --sandbox read-only` stopped before reading the
  diff with `failed to initialize in-process app-server client: Read-only file
  system`.

No `packages/` file changed, so no GYRE trajectory, DINO/tank gate, ORCA2
ladder, or card result is claimed or required. No scientific/configuration
choice was made. The existing sea-ice `unmeasured_features` tuple is unchanged.

## OPEN

1. The operator runs the committed round-90 launcher below. It must admit 80
   frames and prove 20/20 terminal restarts byte-identical to round 83.
2. Once the record admits, name the true rung-0 step-entry state and walk the
   first non-bit boundary in compiled stage order; when that walk reaches WZV,
   measure the B26 two-solve/nemo-literal arm rather than pre-empting order.
3. Build and score the explicit rung-0 card's given-entry ladder, independent
   ladder, and independent month. Rung 1 remains untouched.

Acquisition launcher:
`/tmp/autopilot-orca2-2510514585/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round90_rung0_frames/run.sh`.

## UNVERIFIED

- The optimized repair run, 80-frame admission, and terminal restart identity
  do not exist yet.
- Stage 1 through stage 3, the first model boundary, the rung-0 card ladders,
  and the month score remain unmeasured.
