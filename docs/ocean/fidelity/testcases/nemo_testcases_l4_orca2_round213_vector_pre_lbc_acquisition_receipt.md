# ORCA2 round 213 — OMT-1 vector pre-boundary acquisition

Date: 2026-10-09. Base: `c971cc955`. Preregistration commit:
`95cb270e4`. Acquisition commits: `83e986f38`, `3589eb8d0`, and
`9db1201a6`. Status: **STOPPED_FOR_RECORD**. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round213/`.
No legoESM trajectory was measured. Future comparison numbers remain
separately labelled **independent OMT-1** and **given NEMO's entry OMT-1**.

## Verdict

Round 212's first numerical debt remains unclaimed: on both labels, the same
35 northern-fold cells differ in the slow V forcing and raw V mask, and the
two recorded operands close the raw vector expression only as a pair. NEMO's
executing direct vector update is
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:666-679`. The admitted
stream writes V only after the later boundary exchange at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:721-753`; therefore it
cannot distinguish raw-expression closure from boundary-association closure.

The committed acquisition records the missing target immediately after the
vector update and before bottom drag, depth update, or exchange. It writes one
kt=1/substep-1 file per rank with a self-describing header and three named
arrays: `zv_frc`, `ssvmask`, and `va_pre_lbc`. The checker parses field names
and dimensions from each header, proves complementary exactly-once 148×180
owned coverage, and refuses malformed or non-finite owned payloads.

Passivity is a gate, not an assumption: admission requires all 64 existing
per-stage frame files and both step-8 rank restarts to be byte-identical to the
admitted round-211 OMT-1 run. The operator record does not exist yet, so the
atomic pair, both ladders, and every shared physics gate remain
**UNMEASURED_WITH_SPEC**. No model file or physics statement changed.

## Frozen-prediction disposition

| ID | disposition |
|---|---|
| R213-P1 | **CONFIRMED:** the pinned source, CPP-key, executable, deck-manifest, input-manifest, and complete source-inventory checks pass; the additions-only patch applies with zero removed lines and compiles in fp64. |
| R213-P2 | **UNMEASURED_WITH_SPEC:** the two self-describing per-rank files require operator acquisition. Synthetic parser coverage passes. |
| R213-P3 | **UNMEASURED_WITH_SPEC:** the 64 frame and two terminal-restart byte comparisons require operator acquisition. |
| R213-P4 | **PARTIAL:** layout and source-inventory plants fire in preflight; all eight record/admission plants fire on synthetic records; their real-record executions await acquisition. |
| R213-P5 | **UNMEASURED_WITH_SPEC:** no pre-LBC target exists yet, so no atomic replay or trajectory score was attempted. |

## Validation and review

- Preflight reports `SYNTAX_PROOF_PASS` and
  `ORCA2_ROUND213_VECTOR_PRE_LBC_PREFLIGHT_READY`; its retained log SHA256 is
  `744d4727ab84f74981cd8d259497b878146528d0fa47d2a19faef7a02ce6f179`.
- The layout and closed-source-inventory plants both fire. The focused
  round-211/212/213 and citation-gate battery passes **38/38**.
- The round receipt citation gate passes with two compiled-source citations,
  zero failures, and zero unmapped entries; its rigid-shift plant fires. The
  cumulative default gate passes with 274 citations and zero unmapped entries.
- The single 12-worker `tests/ocean/fidelity` battery collected 3,030 tests and
  reached 99% before its compiler-heavy tail stopped emitting results. It was
  interrupted and is **INCOMPLETE**: 3,000 passed, 7 skipped, 4 failed, and 19
  did not report a terminal result. The four failures are the same registered
  pre-existing/off-scope failures as round 212: the GYRE round-129 spread-floor
  gate, escape-scope ratchet, worktree-stamp ratchet, and SI3 scalar-math
  provenance gate. No second wide battery was started.
- Independent review is unavailable in-sandbox. `codex exec --sandbox
  read-only` exited 1 before reading the diff because its app-server client
  could not create PATH aliases on the read-only filesystem; no review verdict
  is claimed.

No `packages/` file changed, so GYRE, DINO, tanks, rung 0, rung 10, and sea ice
cannot execute changed model code. The shipped ORCA2 card, its six sea-ice
selectors, and its `unmeasured_features` tuple are unchanged.

## OPEN

1. The operator runs
   `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round213_vector_pre_lbc_acquisition/run.sh --run`.
2. The next round admits the two rank files and requires all 64 existing frames
   plus both step-8 restarts to remain byte-identical.
3. Replay the complete recorded slow-V + raw-mask pair against `va_pre_lbc`,
   then score the pair atomically on OMT-1, rung 0, rung 10, GYRE, DINO, and
   tanks under Decision 96. OMT-2 waits for that disposition.

No configuration or sea-ice decision is pending. ASKED choices: Decisions 103
and 109. UNASKED choices: empty.
